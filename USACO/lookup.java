import java.util.*;
import java.io.*;
import java.lang.*;

public class lookup
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("lookup.in"));
      int n = in.nextInt();
      Stack<Integer> stack = new Stack<>();
      int[] cows = new int[n];
      int[] ans = new int[n];
      
      for(int i = 0; i < n; i++)
         cows[i] = in.nextInt();
      
      in.close();
      
      for(int i = 0; i < n; i++)
      {
         if(i == 0)
            stack.push(i);
         else
         {
            while(!stack.isEmpty() && cows[i] > cows[stack.peek()])
               ans[stack.pop()] = i + 1;
            
            stack.push(i);
         }
      }
     
      for(int i = 0; i < n; i++)
         System.out.println(ans[i]);
   }
}