import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.Comparator;
import java.util.Arrays;

public class triangle
{
   private static int[] left;
   private static int[] right;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("triangle.in"));
      int n = in.nextInt();
      
      int[] x = new int[n];
      int[] y = new int[n];
      Triangle[] t = new Triangle[n];
      
      for(int i = 0; i < n; i++)
      {
         x[i] = in.nextInt();
         y[i] = in.nextInt();
         
         t[i] = new Triangle(x[i], y[i], i);
      }
      
      in.close();
      
      Arrays.sort(t, new Compare());
      
      int max = -1, ans = 0;
      for(int i = 0; i < n; i++)
         if(t[i].right > max)
         {
            ans++;
            max = t[i].right;
         }
      
      System.out.println(ans);
   }
   
   public static class Triangle
   {
      public int left;
      public int right;
      public int cid;
      
      public Triangle(int x, int y, int i)
      {
         left = x - y;
         right = x + y;
         cid = i;
      }
   }
   
   public static class Compare implements Comparator<Triangle>
   {
      public int compare(Triangle a, Triangle b)
      {
         if(a.left == b.left)
            return -1 * Integer.compare(a.right, b.right);
         return Integer.compare(a.left, b.left);
      }
   }
}