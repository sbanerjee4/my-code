import java.io.*;
import java.util.*;
import java.lang.*;
public class maze
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("maze.in"));
      int n = in.nextInt(), m = in.nextInt();
      char[][] a = new char[n][m];
      HashMap<Character, ArrayList<Point>> letters = new HashMap<>();
      boolean[][] visited = new boolean[n][m];
      Queue<Integer> qr = new ArrayDeque<>();
      Queue<Integer> qc = new ArrayDeque<>();
      Queue<Integer> qd = new ArrayDeque<>();
      for(int i = 0; i < n; i++)
      {
         char[] temp = in.next().toCharArray();
         a[i] = temp;
         for(int j = 0; j < m; j++)
         {
            if(Character.isLetter(a[i][j]))
            {
               if(!letters.keySet().contains(a[i][j]))
                  letters.put(a[i][j], new ArrayList<Point>());
               letters.get(a[i][j]).add(new Point(i, j));
                              
            } 
            if(a[i][j] == '@')  
            {
               qr.add(i);
               qc.add(j);
               qd.add(0);
               visited[i][j] = true;
            }   
         }
      }
      
      int[] di = {-1, 1, 0, 0};
      int[] dj = {0, 0, -1, 1};
      while(!qr.isEmpty())
      {
         int cr = qr.remove();
         int cc = qc.remove();
         int cd = qd.remove();
         visited[cr][cc] = true;
         if(a[cr][cc] == '=')
         {
            System.out.println(cd);
            System.exit(0);
         }
         
         for(int k = 0; k < 4; k++)
         {
            int r = cr + di[k], c = cc + dj[k];
            if(r >= 0 && r < n && c >= 0 && c < m && !visited[r][c])
            {
               if(Character.isLetter(a[r][c]))
               {
                  char temp = a[r][c];
                  if(letters.get(temp).get(0).x == r && letters.get(temp).get(0).y == c)
                  {
                     qr.add(letters.get(temp).get(1).x);
                     qc.add(letters.get(temp).get(1).y);
                  } 
                  else
                  {
                     qr.add(letters.get(temp).get(0).x);
                     qc.add(letters.get(temp).get(0).y);
                  } 
                  visited[r][c] = true;
                  qd.add(cd + 1);  
               }
               else if(a[r][c] == '.')
               {
                  qr.add(r);
                  qc.add(c);
                  qd.add(cd + 1);
               } 
               else if(a[r][c] == '=')
               {
                  System.out.println(cd + 1);
                  System.exit(0);
               }     
            }
         }
      }
   }
        
   static class Point
   {
      int x;
      int y;
      public Point(int a, int b)
      {
         x = a;
         y = b;
      }
   }
   /*
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("maze.in"));
      int n = in.nextInt(), m = in.nextInt();
      char[][] a = new char[n][m];
      int[][] letters = new int[26][4];
      Queue<Integer> qr = new ArrayDeque<>();
      Queue<Integer> qc = new ArrayDeque<>();
      Queue<Integer> qm = new ArrayDeque<>();
      Map<Character, ArrayList<Point>> map = new TreeMap<>();
      boolean[][] visited = new boolean[n][m];
      for(int i = 0; i < n; i++)
      {
         a[i] = in.next().toCharArray();
         for(int j = 0; j < m; j++)
         {
            if(a[i][j] == '@')
            {
               qr.add(i);
               qc.add(j);
               qm.add(0);
               visited[i][j] = true;
            }
            else if(Character.isLetter(a[i][j]))
            {
               if(map.containsKey(a[i][j]))
               {
                  map.get(a[i][j]).add(new Point(i, j));
               }
               else
               {
                  map.put(a[i][j], new ArrayList<Point>());
                  map.get(a[i][j]).add(new Point(i, j));
               }
            } 
         }
      }
      in.close();
      
      int[] di = {0, 0, -1, 1};
      int[] dj = {-1, 1, 0, 0};
      
      while(!qr.isEmpty())
      {
         int cr = qr.remove();
         int cc = qc.remove();
         int cm = qm.remove();
         if(a[cr][cc] == '=')
         {
            System.out.println(cm);
            System.exit(0);
         }
         for(int i = 0; i < 4; i++)
         {
            int r = cr + di[i];
            int c = cc + dj[i];
            if(r >= 0 && r < n && c >= 0 && c < m)
            {
               if(a[r][c] == '.' && !visited[r][c])
               {
                  qr.add(r);
                  qc.add(c);
                  qm.add(cm + 1);
               }
               else if(Character.isLetter(a[r][c]))
               {
                  if(map.get(a[r][c]).get(0).r == r && map.get(a[r][c]).get(0).c == c)
                  {
                     if(!visited[map.get(a[r][c]).get(1).r][map.get(a[r][c]).get(1).c])
                     {
                        qr.add(map.get(a[r][c]).get(1).r);
                        qc.add(map.get(a[r][c]).get(1).c);
                        qm.add(cm + 1);
                     }  
                  }
                  else
                  {
                     if(!visited[map.get(a[r][c]).get(0).r][map.get(a[r][c]).get(0).c])
                     {
                        qr.add(map.get(a[r][c]).get(0).r);
                        qc.add(map.get(a[r][c]).get(0).c);
                        qm.add(cm + 1);
                     }
                  }
               }
               else if(a[r][c] == '=')
               {
                  System.out.println(cm + 1);
                  System.exit(0);
               }
            }   
         }
      }
   }
   
   static class Point
   {
      public int r;
      public int c;
      public Point(int a, int b)
      {
         r = a;
         c = b;
      }
   }
   */
}